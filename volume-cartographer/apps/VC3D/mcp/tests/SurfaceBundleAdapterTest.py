#!/usr/bin/env python3
"""Regression coverage for TIFXYZ import and registered surface rendering."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("surface_bundle_adapter", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--ink-model-contracts", required=True, type=Path)
    parser.add_argument("--ink-model-root", type=Path)
    arguments = parser.parse_args()
    adapter = load_adapter(arguments.adapter)
    with tempfile.TemporaryDirectory(prefix="vc-surface-bundle-") as temporary:
        root = Path(temporary)
        surface = root / "surface"
        output = root / "output"
        surface.mkdir()
        v, u = np.mgrid[:12, :16]
        tifffile.imwrite(surface / "x.tif", (u + 10).astype(np.float32))
        tifffile.imwrite(surface / "y.tif", (v + 20).astype(np.float32))
        tifffile.imwrite(surface / "z.tif", np.full((12, 16), 40, np.float32))
        tifffile.imwrite(surface / "mask.tif", np.ones((12, 16), np.uint8))
        (surface / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [1, 1]}))
        request = {
            "surface": {"job_id": "job_fixture", "artifact_id": "surface"},
            "coordinate_space": "ct_l0_xyz",
            "uv_region": {"u": 2, "v": 3, "width": 8, "height": 6},
            "normal_padding_voxels": 1,
        }
        imported = adapter.import_surface(request, surface, output)
        assert imported["surface_shape_vu"] == [6, 8]
        assert imported["required_volume_region_xyz"] == {
            "x": 11,
            "y": 22,
            "z": 39,
            "width": 11,
            "height": 9,
            "depth": 4,
        }
        region = imported["required_volume_region_xyz"]
        volume = np.arange(region["depth"] * region["height"] * region["width"], dtype=np.uint16).reshape(
            region["depth"], region["height"], region["width"]
        )
        staging = output / "staging"
        staging.mkdir()
        staged = staging / "staged-volume.npy"
        np.save(staged, volume, allow_pickle=False)
        (staging / "stage-manifest.json").write_text(
            json.dumps({"submitted_region_xyz": {**region, "space": "ct_l0_xyz"}})
        )
        rendered = adapter.render_surface(
            {
                "surface_bundle": imported["surface_bundle"],
                "staged_volume": str(staged),
                "staged_region_xyz": region,
                "volume_source": "fixture",
                "array_path": "0",
                "scale": 0,
                "voxel_spacing": [1.0, 1.0, 1.0],
                "voxel_spacing_unit": "um",
                "voxel_spacing_explicit": True,
                "origin_xyz": [0.0, 0.0, 0.0],
            },
            output,
        )
        assert rendered["coverage_fraction"] == 1.0
        assert rendered["registered_pixels"] == 48
        assert (output / "surface.zarr" / ".zgroup").is_file()
        assert (output / "registered-intensity.npy").is_file()
        assert (output / "registered-intensity.tif").is_file()
        assert (output / "registered-intensity.png").is_file()

        geometry_output = root / "geometry"
        geometry = adapter.geometry_diagnostics(
            {"surface": {"job_id": "registered", "artifact_id": "registered-surface"}}, output, geometry_output
        )
        assert geometry["fold_or_degenerate_cells"] == 0
        assert geometry["p95_normal_change_degrees"] < 0.01
        assert geometry["connected_components"] == 1
        assert geometry["enclosed_holes"] == 0
        assert geometry["global_self_intersection_tested"] is False
        assert (geometry_output / "geometry-diagnostics.zarr" / ".zgroup").is_file()

        alignment_output = root / "alignment"
        alignment = adapter.ct_alignment(
            {
                "surface": {"job_id": "registered", "artifact_id": "registered-surface"},
                "maximum_offset_voxels": 1,
            },
            output,
            alignment_output,
        )
        assert alignment["supported_pixels"] == 48
        assert alignment["support_fraction"] == 1.0
        assert alignment["interpolation"] == "trilinear"
        assert (alignment_output / "ct-alignment.zarr" / ".zgroup").is_file()

        contracts = json.loads(arguments.ink_model_contracts.read_text())
        ink_model_processing = None
        if arguments.ink_model_root is not None:
            timesformer_source = (arguments.ink_model_root / "train_timesformer_og.py").read_text()
            match = re.search(r"def read_image_mask\(fragment_id,start_idx=(\d+),end_idx=(\d+)", timesformer_source)
            assert match and [int(match.group(1)), int(match.group(2))] == [17, 43]
            resnet_metadata = json.loads((arguments.ink_model_root / "metadata.json").read_text())
            assert resnet_metadata["training_hyperparameters"]["model"]["in_chans"] == 62
            assert any(segment.get("layer_range") == [1, 63] for segment in resnet_metadata["segments"].values())
            processing_source = (arguments.ink_model_root / "optimized_inference" / "processing.py").read_text()
            assert "shape=(h, w, c)" in processing_source
            assert "chunks=(chunk_size, chunk_size, 1)" in processing_source
            assert "dtype=np.uint8" in processing_source
            sys.path.insert(0, str(arguments.ink_model_root / "optimized_inference"))
            import processing as ink_model_processing
        model_surface = root / "model-surface"
        model_surface.mkdir()
        model_v, model_u = np.mgrid[:80, :96]
        tifffile.imwrite(model_surface / "x.tif", (model_u + 100).astype(np.float32))
        tifffile.imwrite(model_surface / "y.tif", (model_v + 200).astype(np.float32))
        tifffile.imwrite(model_surface / "z.tif", np.full((80, 96), 100, np.float32))
        tifffile.imwrite(model_surface / "mask.tif", np.ones((80, 96), np.uint8))
        (model_surface / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [1, 1]}))
        undersized_error = None
        try:
            adapter.normal_stack(
                {"surface": {"job_id": "registered", "artifact_id": "registered-surface"},
                 "model_profile": "resnet152-3d-decoder-62"},
                output,
                root / "undersized-stack",
            )
        except ValueError as error:
            undersized_error = str(error)
        assert undersized_error and "height and width" in undersized_error

        low_padding_artifact = root / "model-low-padding"
        low_padding_request = {
            "surface": {"job_id": "job_fixture", "artifact_id": "surface"},
            "coordinate_space": "ct_l0_xyz",
            "uv_region": {"u": 10, "v": 10, "width": 72, "height": 68},
            "normal_padding_voxels": 1,
        }
        adapter.import_surface(low_padding_request, model_surface, low_padding_artifact)
        padding_error = None
        try:
            adapter.normal_stack(
                {"surface": {"job_id": "registered", "artifact_id": "registered-surface"},
                 "model_profile": "resnet152-3d-decoder-62"},
                low_padding_artifact,
                root / "low-padding-stack",
            )
        except ValueError as error:
            padding_error = str(error)
        assert padding_error and "requires normal padding" in padding_error

        model_artifact = root / "model-registered"
        model_request = {
            "surface": {"job_id": "job_fixture", "artifact_id": "surface"},
            "coordinate_space": "ct_l0_xyz",
            "uv_region": {"u": 10, "v": 10, "width": 72, "height": 68},
            "normal_padding_voxels": 32,
        }
        model_import = adapter.import_surface(model_request, model_surface, model_artifact)
        model_region = model_import["required_volume_region_xyz"]
        zz, yy, xx = np.mgrid[
            model_region["z"] : model_region["z"] + model_region["depth"],
            model_region["y"] : model_region["y"] + model_region["height"],
            model_region["x"] : model_region["x"] + model_region["width"],
        ]
        model_volume = ((zz * 3 + yy * 5 + xx * 7) % 256).astype(np.uint8)
        model_staging = model_artifact / "staging"
        model_staging.mkdir()
        np.save(model_staging / "staged-volume.npy", model_volume, allow_pickle=False)
        (model_staging / "stage-manifest.json").write_text(
            json.dumps({"submitted_region_xyz": {**model_region, "space": "ct_l0_xyz"}})
        )

        import zarr

        rendered_stacks = {}
        for profile in ("timesformer-26", "resnet152-3d-decoder-62"):
            profile_output = root / profile
            rendered = adapter.normal_stack(
                {
                    "surface": {"job_id": "registered", "artifact_id": "registered-surface"},
                    "model_profile": profile,
                    "layer_step_voxels": 1.0,
                    "reverse_layers": False,
                },
                model_artifact,
                profile_output,
            )
            expected = contracts["profiles"][profile]
            assert rendered["shape_hwc"] == [68, 72, expected["channels"]]
            assert rendered["dtype"] == "uint8"
            assert rendered["axes"] == ["v", "u", "normal_depth"]
            assert rendered["source_layer_range"] == expected["source_layer_range"]
            assert rendered["offsets_voxels"][0] == expected["offsets_voxels"][0]
            assert rendered["offsets_voxels"][-1] == expected["offsets_voxels"][1]
            stack = np.asarray(zarr.open(str(profile_output / "surface-volume.zarr"), mode="r"))
            assert stack.shape == (68, 72, expected["channels"])
            assert stack.dtype == np.uint8
            assert zarr.open(str(profile_output / "surface-volume.zarr"), mode="r").chunks[-1] == 1
            assert len(list((profile_output / "layers").glob("*.tif"))) == expected["channels"]
            assert len(rendered["layer_hashes"]) == expected["channels"]
            assert rendered["support_fraction"] == 1.0
            # Plane normal is +Z, so integer normal offsets must match direct fixture sampling.
            first_xyz = np.array([110, 210, 100])
            expected_values = [int((first_xyz[2] + offset) * 3 + first_xyz[1] * 5 + first_xyz[0] * 7) % 256
                               for offset in range(expected["offsets_voxels"][0], expected["offsets_voxels"][1] + 1)]
            assert stack[0, 0].tolist() == expected_values
            if ink_model_processing is not None:
                model_output = root / f"{profile}-model-prepare.zarr"
                ink_model_processing.create_surface_volume_zarr(
                    [str(path) for path in sorted((profile_output / "layers").glob("*.tif"))],
                    str(model_output),
                    chunk_size=64,
                    max_workers=2,
                    use_compression=False,
                )
                model_stack = np.asarray(zarr.open(str(model_output), mode="r"))
                assert model_stack.shape == stack.shape
                assert model_stack.dtype == np.uint8
                assert np.array_equal(model_stack, stack)
            rendered_stacks[profile] = stack

        reverse_output = root / "resnet152-3d-decoder-62-reversed"
        reversed_manifest = adapter.normal_stack(
            {
                "surface": {"job_id": "registered", "artifact_id": "registered-surface"},
                "model_profile": "resnet152-3d-decoder-62",
                "layer_step_voxels": 1.0,
                "reverse_layers": True,
            },
            model_artifact,
            reverse_output,
        )
        reversed_stack = np.asarray(zarr.open(str(reverse_output / "surface-volume.zarr"), mode="r"))
        assert np.array_equal(reversed_stack, rendered_stacks["resnet152-3d-decoder-62"][..., ::-1])
        assert reversed_manifest["offsets_voxels"] == list(reversed(range(-31, 31)))
    print("SurfaceBundleAdapterTest passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
