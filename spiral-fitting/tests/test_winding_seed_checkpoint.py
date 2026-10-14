"""Real ray reconstruction accepts both generations of Spiral config keys."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from config import Config
from sample_spiral import get_spiral_yxs
from transforms import SpiralAndTransform
from umbilicus import json_umbilicus_z_to_yx


SOURCE = (Path(__file__).resolve().parents[2] / "vesuvius/src/vesuvius"
          / "neural_tracing/winding_models/infer_winding_volume.py")
spec = importlib.util.spec_from_file_location("winding_seed_inference", SOURCE)
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)

MODEL_KEYS = (
    "num_flow_integration_steps", "flow_integration_solver", "num_flow_stages",
    "flow_bounds_z_margin", "flow_bounds_radius", "flow_voxel_resolution",
    "flow_field_type", "gap_expander_logit_resolution", "gap_expander_num_windings",
    "gap_expander_lr_scale", "linear_z_resolution", "initial_dr_per_winding",
)


@pytest.mark.parametrize("format", ["current", "legacy", "legacy_default_stages", "mixed"])
def test_checkpoint_config_reconstructs_fitted_ray_positions(tmp_path, format):
    config = Config().as_dict()
    config.update(
        model_initial_dr_per_winding=10., model_flow_field_type="cartesian",
        model_flow_voxel_resolution=16, model_linear_z_resolution=32,
        model_gap_expander_num_windings=14, model_gap_expander_capacity_windings=14,
        model_gap_expander_logit_resolution=24, model_gap_expander_lr_scale=100.,
        model_flow_bounds_z_margin=0, model_flow_bounds_radius=160,
        model_num_flow_integration_steps=24, model_num_flow_stages=1,
        model_flow_integration_solver="rk4",
    )
    z_begin, z_end, spacing = 3100, 3360, 16
    umbilicus = Path(__file__).parent / "data/umbilicus.json"
    zs = np.arange(z_begin, z_end)
    centre = json_umbilicus_z_to_yx(str(umbilicus))
    model = SpiralAndTransform(
        24, "rk4", torch.tensor([z_begin, -160, -160]),
        torch.tensor([z_end, 160, 160]),
        torch.from_numpy(np.c_[zs, centre(zs)].astype(np.float32)), config,
    ).eval()
    # Nonzero learned parameters must survive the checkpoint reconstruction.
    with torch.no_grad():
        model.linear_logits[:, 0, 0] = .001
    saved_config = dict(config)
    if format.startswith("legacy"):
        for key in MODEL_KEYS:
            saved_config[key] = saved_config.pop(f"model_{key}")
        if format == "legacy_default_stages":
            saved_config.pop("num_flow_stages")
    elif format == "mixed":
        saved_config.update({key: "retired-value-must-not-win" for key in MODEL_KEYS})
    checkpoint = tmp_path / "fit.ckpt"
    torch.save({
        "cfg": saved_config, "spiral_and_transform": model.state_dict(),
        "z_begin": z_begin, "z_end": z_end, "spiral_outward_sense": "CW",
    }, checkpoint)
    args = inference.argparse.Namespace(
        fit_checkpoint=str(checkpoint), umbilicus=str(umbilicus), z_range=None,
        winding_range=(10, 12), winding_step=1, seed_spacing=spacing,
    )
    rays = inference.build_seed_rays(args, device="cpu")
    with torch.inference_mode():
        z_rows = torch.arange(z_begin, z_end, spacing, dtype=torch.float32)
        yxs = get_spiral_yxs(13, model.get_dr_per_winding(), spacing, True)
        transform = model.get_slice_to_spiral_transform()
        for winding in (10, 11, 12):
            grid = torch.cat([
                z_rows[:, None, None].expand(-1, len(yxs[winding]), 1),
                yxs[winding][None].expand(len(z_rows), -1, 2),
            ], dim=-1)
            expected = transform.inv(grid.reshape(-1, 3)).numpy()[:, ::-1]
            np.testing.assert_allclose(
                rays["seed_xyz"][rays["seed_winding"] == winding], expected,
                rtol=0, atol=1e-3,
            )
    np.testing.assert_allclose(
        np.linalg.norm(rays["direction_xyz"], axis=1), 1., atol=1e-6,
    )
    assert np.isfinite(rays["direction_xyz"]).all()
    assert (rays["seed_xyz"][:, 2] >= z_begin).all()
    assert (rays["seed_xyz"][:, 2] < z_end).all()


@pytest.mark.parametrize("format", ["current", "legacy", "mixed"])
def test_mesh_seed_defaults_use_checkpoint_model_keys(tmp_path, format):
    import json
    from PIL import Image

    config = {"output_first_winding": 10,
              "model_gap_expander_num_windings": 14,
              "model_initial_dr_per_winding": 10.}
    if format == "legacy":
        config["gap_expander_num_windings"] = config.pop("model_gap_expander_num_windings")
        config["initial_dr_per_winding"] = config.pop("model_initial_dr_per_winding")
    elif format == "mixed":
        config.update(gap_expander_num_windings=12, initial_dr_per_winding=99.)
    checkpoint = tmp_path / "fit.ckpt"
    torch.save({"cfg": config, "z_begin": 3100, "z_end": 3200}, checkpoint)
    umbilicus = Path(__file__).parent / "data/umbilicus.json"
    centre = json_umbilicus_z_to_yx(str(umbilicus))
    for winding in (10, 11, 12, 13):
        output = tmp_path / f"w{winding:03d}_spliced"
        output.mkdir()
        zz, theta = np.meshgrid(np.arange(3100, 3200, 10),
                               np.linspace(0, 2 * np.pi, 16, endpoint=False), indexing="ij")
        yx = centre(zz)
        arrays = {"z": zz, "x": yx[..., 1] + winding * 10 * np.cos(theta),
                  "y": yx[..., 0] + winding * 10 * np.sin(theta)}
        for axis, values in arrays.items():
            Image.fromarray(values.astype(np.float32)).save(output / f"{axis}.tif")
        (output / "meta.json").write_text(json.dumps({
            "uuid": output.name, "scale": [.1, .1], "format": "tifxyz", "type": "seg",
        }), encoding="utf-8")
    args = inference.argparse.Namespace(
        fit_checkpoint=str(checkpoint), umbilicus=str(umbilicus), z_range=None,
        winding_range=None, winding_step=1, seed_spacing=10, meshes_dir=str(tmp_path),
    )
    rays = inference.build_seed_rays_from_meshes(args)
    assert rays["dr_per_winding"] == 10.
    np.testing.assert_array_equal(rays["seed_winding"], np.repeat([10, 11, 12, 13], 160))
    np.testing.assert_allclose(np.linalg.norm(rays["direction_xyz"], axis=1), 1., atol=1e-6)
